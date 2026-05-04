"""Video player entry point for Pupil Capture recordings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from eye_synth.input.recording import Recording
from eye_synth.output import (
    ColorConsoleSink, DEFAULT_OVERLAY, MultiSink, PureDataSink, draw_overlay,
)
from eye_synth.patches import Patch, load_patch
from eye_synth.signals.bus import OutputBus
from eye_synth.signals.pipeline import Pipeline



class GazeVideoPlayer:
    """Interactive video player that displays gaze position overlay."""

    def __init__(
        self,
        recording: Recording,
        pipeline: Pipeline | None = None,
        output: MultiSink | None = None,
        outputs: OutputBus | None = None,
        patch: Patch | None = None,
        region_size: int = 50,
        show_overlay: bool = False,
    ):
        self.recording = recording
        self.pipeline = pipeline
        self.output = output
        self.region_size = region_size
        self.show_overlay = show_overlay
        self.frame_index = 0
        self.playing = False
        self.playback_speed = 1.0

        # Display settings
        self.gaze_radius = 20
        self.gaze_color = (0, 255, 0)
        self.gaze_thickness = 2
        self.low_confidence_color = (0, 165, 255)
        self.confidence_threshold = 0.6
        self.window_name = f"Gaze Player - {recording.recording_name}"

        self._outputs = outputs if outputs is not None else OutputBus()
        self._patch = patch if patch is not None else load_patch("TNC_v1")
        self._overlay_cfg = getattr(self._patch, 'overlay', DEFAULT_OVERLAY)

        # Cached color reading for overlay — persists across pipeline.reset() on pause
        self._last_color_reading = None

        # Blink/flutter display state
        self._blink_flash_until = 0.0
        self._last_blink_label: str | None = None
        self._flutter_flash_until = 0.0
        self._last_flutter_event = None

        # Note display state
        self._note_flash_until = 0.0
        self._last_displayed_note: int | None = None
        self._dry_outputs = OutputBus()  # no PD sink — used when paused

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
            "UP       - Step forward 1 frame",
            "DOWN     - Step back 1 frame",
            "LEFT     - Back 30 frames",
            "RIGHT    - Forward 30 frames",
            "HOME     - Go to start",
            "END      - Go to end",
            "O        - Decrease speed",
            "P        - Increase speed",
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
        self._last_flutter_event = None
        if self.pipeline is not None:
            self.pipeline.reset()
        self._patch.reset()

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

            # --- Analysis (always) and patch dispatch (only when playing) ---
            if self.pipeline is not None:
                world_ts = self.recording.get_frame_timestamp(self.frame_index)
                gaze = self.recording.get_gaze_for_frame(self.frame_index)
                blink = self.recording.get_blink_at_timestamp(world_ts)
                flutter = self.recording.get_flutter_at_timestamp(world_ts)
                fixation = self.recording.get_fixation_for_frame(self.frame_index)

                gaze_px = None
                if gaze is not None:
                    gaze_px = self.recording.gaze_to_pixel(
                        gaze, current_frame.shape[1], current_frame.shape[0]
                    )

                signals = self.pipeline.process_recording_frame(
                    current_frame, gaze, blink, flutter, fixation, world_ts,
                    gaze_px=gaze_px,
                    min_confidence=self.confidence_threshold,
                )
                if signals.has_env_reading and self.pipeline.last_color_reading is not None:
                    self._last_color_reading = self.pipeline.last_color_reading
                    if self.output is not None:
                        self.output.emit(self._last_color_reading)

                active_outputs = self._outputs if self.playing else self._dry_outputs
                self._patch.update(signals, active_outputs)
                _current_note = getattr(self._patch, "current_midi_note", None)
                if _current_note != self._last_displayed_note:
                    self._last_displayed_note = _current_note
                    self._note_flash_until = world_ts + 0.4

            # --- Draw overlays ---
            display_frame = current_frame.copy()
            if self.show_overlay:
                _gaze = self.recording.get_gaze_for_frame(self.frame_index)
                _gaze_px = None
                _confidence = 0.0
                if _gaze is not None:
                    _gaze_px = self.recording.gaze_to_pixel(
                        _gaze, display_frame.shape[1], display_frame.shape[0]
                    )
                    _confidence = _gaze.confidence

                _eye_frame = self.recording.get_eye_frame_for_world_frame(self.frame_index)
                _world_ts = self.recording.get_frame_timestamp(self.frame_index)

                _blink_rec = self.recording.get_blink_at_timestamp(_world_ts)
                if _blink_rec is not None:
                    self._blink_flash_until = _world_ts + 0.2
                    self._last_blink_label = (
                        f"{_blink_rec.blink_type.value.upper()} {_blink_rec.duration_ms:.0f}ms"
                        if _blink_rec.duration_ms >= 0 else "BLINK"
                    )
                _is_blinking = _world_ts < self._blink_flash_until

                _flutter_rec = self.recording.get_flutter_at_timestamp(_world_ts)
                if _flutter_rec is not None:
                    self._flutter_flash_until = _world_ts + 0.2
                    self._last_flutter_event = _flutter_rec
                _is_flutter = _world_ts < self._flutter_flash_until
                _flutter_label = (
                    f"FLUTTER {self._last_flutter_event.blink_count} blinks"
                    if _is_flutter and self._last_flutter_event is not None else None
                )

                _note_flash = _world_ts < self._note_flash_until

                draw_overlay(
                    display_frame, self._overlay_cfg,
                    gaze_px=_gaze_px,
                    confidence=_confidence,
                    region_size=self.region_size,
                    color_reading=self._last_color_reading,
                    eye_frame=_eye_frame,
                    is_blink=_is_blinking,
                    blink_label=self._last_blink_label if _is_blinking else None,
                    is_flutter=_is_flutter,
                    flutter_label=_flutter_label,
                    blink_count=len(self.recording.blink_data),
                    flutter_count=len(self.recording.flutter_data),
                    current_note=self._last_displayed_note,
                    note_flash=_note_flash,
                )
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
            elif key == 82 or key == 0:   # Up arrow — step forward 1 frame
                self.playing = False
                self.frame_index = min(total_frames - 1, self.frame_index + 1)
                need_seek = True
            elif key == 84 or key == 1:   # Down arrow — step back 1 frame
                self.playing = False
                self.frame_index = max(0, self.frame_index - 1)
                need_seek = True
            elif key == 81 or key == 2:   # Left arrow — back 30 frames
                self.playing = False
                self.frame_index = max(0, self.frame_index - 30)
                need_seek = True
            elif key == 83 or key == 3:   # Right arrow — forward 30 frames
                self.playing = False
                self.frame_index = min(total_frames - 1, self.frame_index + 30)
                need_seek = True
            elif key == 80:               # Home
                self.frame_index = 0
                need_seek = True
            elif key == 87:               # End
                self.frame_index = total_frames - 1
                need_seek = True
            elif key == ord("o"):
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
            elif key == ord("p"):
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
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Video player with gaze overlay for Pupil Capture recordings"
    )
    parser.add_argument("recording_path", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable colour/brightness overlay")
    parser.add_argument("--patch", type=str, default="TNC_v1",
                        help="Patch to use for mapping signals to outputs")
    parser.add_argument("--pd-host", type=str, default="127.0.0.1",
                        help="Pure Data host (default: 127.0.0.1)")
    parser.add_argument("--pd-port", type=int, default=9001,
                        help="Pure Data port (default: 9001)")
    args = parser.parse_args()

    recording_path = Path(args.recording_path)
    if not recording_path.exists():
        print(f"Error: Recording not found: {recording_path}")
        sys.exit(1)

    show_overlay = not args.no_overlay
    pipeline = Pipeline()
    pd_sink = PureDataSink(host=args.pd_host, port=args.pd_port)
    outputs = OutputBus(pd_sink)

    patch = load_patch(args.patch)

    output: MultiSink | None = None
    if args.verbose:
        output = MultiSink()
        output.add_sink(ColorConsoleSink(verbose=True))

    try:
        with Recording(recording_path) as recording:
            info = recording.get_info()
            print(f"Loaded recording: {info.recording_name}")
            print(f"  Duration: {info.duration_s:.1f}s  Frames: {info.frame_count}")
            print(f"  Resolution: {info.world_resolution[0]}x{info.world_resolution[1]}")
            print(f"  Gaze: {info.gaze_count}  Fixations: {info.fixation_count}  Blinks: {info.blink_count}")
            print(f"  Patch: {args.patch}")

            player = GazeVideoPlayer(
                recording,
                pipeline=pipeline,
                output=output,
                outputs=outputs,
                patch=patch,
                show_overlay=show_overlay,
            )
            player.run()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    finally:
        patch.shutdown(outputs)
        pd_sink.close()


if __name__ == "__main__":
    main()
