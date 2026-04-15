"""Main entry point for the Pupil Color-to-Music Tracker."""

import argparse
import sys
import time

import cv2
import numpy as np

from pupil_tracker.input.live import PupilCaptureClient
from pupil_tracker.output import (
    ColorConsoleSink, MultiSink, PureDataSink,
    apply_gamma, build_gamma_lut,
    draw_brightness_bar, draw_color_info, draw_eye_panel,
    draw_gaze_crosshair, draw_region_box,
)
from pupil_tracker.patches import load_patch
from pupil_tracker.signals.bus import OutputBus, SignalBus
from pupil_tracker.signals.env_color import ColorAnalyzer, ColorReading, FrameProcessor
from pupil_tracker.signals.env_scene_change import SceneChangeDetector
from pupil_tracker.signals.eye_blinks import StreamingBlinkTracker
from pupil_tracker.signals.eye_gaze import GazeVelocityTracker


def run_tracker(
    host: str = "127.0.0.1",
    port: int = 50020,
    region_size: int = 50,
    smoothing: int = 5,
    show_video: bool = True,
    verbose: bool = False,
    pd: bool = False,
    pd_host: str = "127.0.0.1",
    pd_port: int = 9001,
    note_stability: int = 2,
    octave_stability: int = 3,
    octave_threshold: float = 0.5,
    gamma: float = 1.0,
    patch_name: str = "color_music",
) -> None:
    """Run the color-to-music tracker."""
    # --- Detection pipeline ---
    processor = FrameProcessor(region_size=region_size)
    analyzer = ColorAnalyzer(
        smoothing_window=smoothing,
        note_stability_frames=note_stability,
        octave_stability_frames=octave_stability,
        octave_stability_threshold=octave_threshold,
    )
    blink_tracker = StreamingBlinkTracker()
    scene_detector = SceneChangeDetector()
    gaze_vel = GazeVelocityTracker()
    gamma_lut = build_gamma_lut(gamma)

    # --- Signal bus and patch ---
    signals = SignalBus()
    pd_sink: PureDataSink | None = PureDataSink(host=pd_host, port=pd_port) if pd else None
    outputs = OutputBus(pd_sink)
    patch = load_patch(patch_name)

    # --- Console output (continuous, separate from patch) ---
    console_output = MultiSink()
    console_output.add_sink(ColorConsoleSink(verbose=verbose))

    # --- Display state (overlay only, not sent to outputs) ---
    blink_flash_until = 0.0
    last_blink_label: str | None = None
    flutter_flash_until = 0.0
    last_flutter_label: str | None = None
    latest_eye_frame: np.ndarray | None = None
    last_reading: ColorReading | None = None

    print("=" * 60)
    print("Pupil Color-to-Music Tracker")
    print("=" * 60)
    print(f"  Host: {host}:{port}")
    print(f"  Region size: {region_size}px")
    print(f"  Smoothing window: {smoothing} frames")
    print(f"  Video display: {'enabled' if show_video else 'disabled'}")
    print(f"  Patch: {patch_name}")
    if pd:
        print(f"  Pure Data (FUDI): {pd_host}:{pd_port}")
    if gamma != 1.0:
        print(f"  Gamma correction: {gamma}")
    print("=" * 60)
    print("Press 'q' in the video window or Ctrl+C to stop.")
    print()

    try:
        with PupilCaptureClient(host=host, port=port) as client:
            for message in client.stream_realtime():
                now = time.monotonic()
                signals.clear_events()
                signals.timestamp = now

                # --- Update gaze ---
                if message.gaze is not None:
                    if processor.update_gaze(message.gaze):
                        signals.eye.confidence = message.gaze.confidence
                        signals.eye.norm_pos = message.gaze.norm_pos

                # --- New fixation ---
                if message.fixation is not None:
                    signals.eye.fixation_id = message.fixation.id

                # --- Blink events ---
                if message.blink is not None:
                    b = message.blink
                    blink_event, _ = blink_tracker.update(
                        b.blink_type, b.timestamp, b.confidence
                    )
                    if blink_event is not None:
                        signals.eye.blink = blink_event
                        blink_flash_until = now + 0.3
                        if blink_event.duration_ms >= 0:
                            last_blink_label = (
                                f"{blink_event.blink_type.value.upper()} "
                                f"{blink_event.duration_ms:.0f}ms"
                            )
                        else:
                            last_blink_label = "BLINK"
                    if blink_tracker.is_flutter_active:
                        last_flutter_label = (
                            f"FLUTTER {blink_tracker.active_flutter_blink_count} blinks"
                        )

                # --- Flutter timeout ---
                flutter_event = blink_tracker.tick(now)
                if flutter_event is not None:
                    signals.eye.flutter = flutter_event
                    flutter_flash_until = now + 0.3
                    last_flutter_label = f"FLUTTER {flutter_event.blink_count} blinks"

                # --- Sync blink/flutter state to bus ---
                signals.eye.is_eyes_closed = blink_tracker.is_eyes_closed
                signals.eye.is_flutter_active = blink_tracker.is_flutter_active
                signals.eye.flutter_blink_count = blink_tracker.active_flutter_blink_count
                signals.eye.total_blinks = blink_tracker.blink_count
                signals.eye.total_flutters = blink_tracker.flutter_count

                # --- Eye frame for display ---
                if message.eye_frame is not None:
                    latest_eye_frame = message.eye_frame.data

                # --- Frame processing ---
                if message.frame is not None:
                    processor.update_frame(message.frame)
                    signals.frame_width = message.frame.width
                    signals.frame_height = message.frame.height

                    # Gamma correction
                    frame_data = message.frame.data
                    if gamma != 1.0:
                        frame_data = apply_gamma(frame_data, gamma_lut)
                        processor._last_frame = message.frame.__class__(
                            timestamp=message.frame.timestamp,
                            width=message.frame.width,
                            height=message.frame.height,
                            data=frame_data,
                            topic=message.frame.topic,
                        )

                    # Scene change (full frame)
                    signals.env.scene_change = scene_detector.update(frame_data)

                    # Gaze region analysis
                    gaze_region = processor.extract_region()
                    if gaze_region is not None:
                        color_reading = analyzer.analyze(gaze_region)
                        last_reading = color_reading

                        # Populate env signals
                        signals.env.hue = color_reading.smoothed_hue
                        signals.env.hue_normalized = color_reading.smoothed_hue / 179.0
                        signals.env.saturation = color_reading.saturation
                        signals.env.brightness = color_reading.smoothed_brightness
                        signals.env.brightness_normalized = (
                            color_reading.smoothed_brightness / 255.0
                        )
                        signals.env.note = color_reading.note
                        signals.env.octave = color_reading.octave
                        signals.env.midi_note = color_reading.midi_note
                        signals.env.raw_midi_note = color_reading.raw_midi_note
                        signals.has_env_reading = True

                        # Gaze velocity and pixel position
                        if processor.last_gaze is not None:
                            gx, gy = processor.norm_to_pixel(
                                processor.last_gaze.norm_pos[0],
                                processor.last_gaze.norm_pos[1],
                                message.frame.width,
                                message.frame.height,
                            )
                            signals.eye.px_pos = (gx, gy)
                            gaze_vel.update(
                                processor.last_gaze.norm_pos,
                                processor.last_gaze.timestamp,
                                message.frame.width,
                                message.frame.height,
                            )
                            signals.eye.velocity_px_s = gaze_vel.velocity

                        console_output.emit(color_reading)

                    # --- Display overlay ---
                    if show_video:
                        frame = frame_data.copy()

                        if processor.last_gaze is not None:
                            gx, gy = processor.norm_to_pixel(
                                processor.last_gaze.norm_pos[0],
                                processor.last_gaze.norm_pos[1],
                                message.frame.width,
                                message.frame.height,
                            )
                            draw_gaze_crosshair(frame, gx, gy, processor.last_gaze.confidence)
                            draw_region_box(frame, gx, gy, region_size, processor.last_gaze.confidence)

                        if last_reading is not None:
                            draw_brightness_bar(frame, last_reading.smoothed_brightness)
                            draw_color_info(frame, last_reading)

                        is_blink = now < blink_flash_until
                        is_flutter = now < flutter_flash_until or blink_tracker.is_flutter_active
                        draw_eye_panel(
                            frame,
                            latest_eye_frame,
                            is_blink=is_blink,
                            blink_label=last_blink_label if is_blink else None,
                            is_flutter=is_flutter,
                            flutter_label=last_flutter_label if is_flutter else None,
                            blink_count=blink_tracker.blink_count,
                            flutter_count=blink_tracker.flutter_count,
                        )

                        cv2.imshow("Pupil Color-to-Music", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # --- Dispatch to patch ---
                patch.update(signals, outputs)

    except ConnectionError as e:
        print(f"\n[ERROR] Could not connect to Pupil Capture: {e}")
        print("Make sure Pupil Capture is running and Frame Publisher is enabled.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        console_output.close()
        if pd_sink is not None:
            pd_sink.send("confidence", 1.0)
            pd_sink.send("am_lfo", 0)
            pd_sink.close()
        if show_video:
            cv2.destroyAllWindows()

    print("[INFO] Tracker stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stream gaze data from Pupil Core and map colors to musical notes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50020)
    parser.add_argument("--region-size", type=int, default=50,
                        help="Size of the gaze region to analyze (pixels)")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="Number of frames to average for smoothing")
    parser.add_argument("--no-video", action="store_true", help="Disable video display")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (< 1.0 brightens, > 1.0 darkens)")
    parser.add_argument("--patch", type=str, default="color_music",
                        help="Patch to use for mapping signals to outputs")

    stability_group = parser.add_argument_group("Stability tuning")
    stability_group.add_argument("--note-stability", type=int, default=2)
    stability_group.add_argument("--octave-stability", type=int, default=3)
    stability_group.add_argument("--octave-threshold", type=float, default=0.5)

    pd_group = parser.add_argument_group("Pure Data output")
    pd_group.add_argument("--pd", action="store_true",
                          help="Send to Pure Data via FUDI protocol (TCP)")
    pd_group.add_argument("--pd-host", type=str, default="127.0.0.1")
    pd_group.add_argument("--pd-port", type=int, default=9001)

    args = parser.parse_args()

    run_tracker(
        host=args.host,
        port=args.port,
        region_size=args.region_size,
        smoothing=args.smoothing,
        show_video=not args.no_video,
        verbose=args.verbose,
        pd=args.pd,
        pd_host=args.pd_host,
        pd_port=args.pd_port,
        note_stability=args.note_stability,
        octave_stability=args.octave_stability,
        octave_threshold=args.octave_threshold,
        gamma=args.gamma,
        patch_name=args.patch,
    )


if __name__ == "__main__":
    main()
