"""Main entry point for the Pupil Color-to-Music Tracker."""

import argparse
import sys
import time

import cv2
import numpy as np

from pupil_tracker.analyzer import ColorAnalyzer, ColorReading, NoteEvent, NoteGate
from pupil_tracker.client import PupilCaptureClient
from pupil_tracker.eye_events import StreamingBlinkTracker
from pupil_tracker.output import (
    ColorConsoleSink,
    MultiSink,
    PureDataSink,
)
from pupil_tracker.overlay import (
    apply_gamma,
    build_gamma_lut,
    draw_brightness_bar,
    draw_color_info,
    draw_eye_panel,
    draw_gaze_crosshair,
    draw_region_box,
)
from pupil_tracker.processor import FrameProcessor


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
) -> None:
    """Run the color-to-music tracker.

    Args:
        host: Pupil Capture host address.
        port: Pupil Capture control port.
        region_size: Size of the gaze region to analyze.
        smoothing: Number of frames to average for smoothing.
        show_video: Whether to display the video feed.
        verbose: Whether to print verbose console output.
        pd: Enable Pure Data output via FUDI protocol.
        pd_host: Pure Data host address.
        pd_port: Pure Data FUDI port.
        note_stability: Frames for note stability.
        octave_stability: Frames for octave stability.
        octave_threshold: Agreement threshold for octave changes (0-1).
        gamma: Gamma correction value (< 1.0 brightens, > 1.0 darkens).
    """
    # Initialize components
    processor = FrameProcessor(region_size=region_size)
    analyzer = ColorAnalyzer(
        smoothing_window=smoothing,
        note_stability_frames=note_stability,
        octave_stability_frames=octave_stability,
        octave_stability_threshold=octave_threshold,
    )

    # Gamma correction
    gamma_lut = build_gamma_lut(gamma)

    # Set up output sinks
    output = MultiSink()
    output.add_sink(ColorConsoleSink(verbose=verbose))

    # Set up Pure Data output for note events
    pd_sink: PureDataSink | None = None
    if pd:
        pd_sink = PureDataSink(host=pd_host, port=pd_port)

    # Content-based note triggering
    note_gate = NoteGate()

    # Streaming blink tracker (classifies blinks, detects flutter)
    blink_tracker = StreamingBlinkTracker()

    # Eye event display state
    blink_flash_until = 0.0
    last_blink_label: str | None = None
    flutter_flash_until = 0.0
    last_flutter_label: str | None = None

    # Latest eye frame for display
    latest_eye_frame: np.ndarray | None = None

    print("=" * 60)
    print("Pupil Color-to-Music Tracker")
    print("=" * 60)
    print(f"  Host: {host}:{port}")
    print(f"  Region size: {region_size}px")
    print(f"  Smoothing window: {smoothing} frames")
    print(f"  Video display: {'enabled' if show_video else 'disabled'}")
    print("  Mode: COLOR → MUSIC (content-based note triggering)")
    if pd:
        print(f"  Pure Data (FUDI): {pd_host}:{pd_port}")
    if gamma != 1.0:
        print(f"  Gamma correction: {gamma}")
    print("=" * 60)
    print("Press 'q' in the video window or Ctrl+C to stop.")
    print()

    last_reading: ColorReading | None = None

    try:
        with PupilCaptureClient(host=host, port=port) as client:
            for message in client.stream_realtime():
                now = time.monotonic()

                # Update gaze (may come with frame or alone)
                if message.gaze is not None:
                    processor.update_gaze(message.gaze)

                # New fixation = new object of interest
                if message.fixation is not None:
                    note_gate.new_fixation(message.fixation.id)

                # Feed blink events to tracker
                if message.blink is not None:
                    b = message.blink
                    blink_event, flutter_event = blink_tracker.update(
                        b.blink_type, b.timestamp, b.confidence
                    )
                    if blink_event is not None:
                        blink_flash_until = now + 0.3
                        if blink_event.duration_ms >= 0:
                            last_blink_label = (
                                f"{blink_event.blink_type.value.upper()} "
                                f"{blink_event.duration_ms:.0f}ms"
                            )
                        else:
                            last_blink_label = "BLINK"

                    if flutter_event is not None:
                        flutter_flash_until = now + 0.3
                        last_flutter_label = (
                            f"FLUTTER {flutter_event.blink_count} blinks"
                        )

                # Store latest eye frame for display
                if message.eye_frame is not None:
                    latest_eye_frame = message.eye_frame.data

                # Process on new frame (real-time, no buffering)
                if message.frame is not None:
                    processor.update_frame(message.frame)

                    # Apply gamma correction before analysis
                    frame_data = message.frame.data
                    if gamma != 1.0:
                        frame_data = apply_gamma(frame_data, gamma_lut)
                        # Update processor's frame with gamma-corrected data
                        processor._last_frame = message.frame.__class__(
                            timestamp=message.frame.timestamp,
                            width=message.frame.width,
                            height=message.frame.height,
                            data=frame_data,
                            topic=message.frame.topic,
                        )

                    # Extract gaze region and analyze for display
                    gaze_region = processor.extract_region()
                    if gaze_region is not None:
                        color_reading = analyzer.analyze(gaze_region)
                        output.emit(color_reading)
                        last_reading = color_reading

                        # Content-based note triggering (suppressed during flutter)
                        if (
                            pd_sink is not None
                            and not blink_tracker.is_flutter_active
                            and note_gate.update(color_reading.midi_note, color_reading.raw_midi_note)
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
                            pd_sink.emit(note_event)

                    # Display video with overlay
                    if show_video:
                        frame = frame_data.copy()

                        # Draw gaze crosshair and region box
                        if processor.last_gaze is not None:
                            gx, gy = processor.norm_to_pixel(
                                processor.last_gaze.norm_pos[0],
                                processor.last_gaze.norm_pos[1],
                                message.frame.width,
                                message.frame.height,
                            )
                            draw_gaze_crosshair(
                                frame, gx, gy,
                                processor.last_gaze.confidence,
                            )
                            draw_region_box(
                                frame, gx, gy,
                                region_size,
                                processor.last_gaze.confidence,
                            )

                        # Draw color/brightness info
                        if last_reading is not None:
                            draw_brightness_bar(
                                frame,
                                last_reading.smoothed_brightness,
                            )
                            draw_color_info(frame, last_reading)

                        # Draw eye camera panel with event indicators
                        is_blink = now < blink_flash_until
                        is_flutter = (
                            now < flutter_flash_until
                            or blink_tracker.is_flutter_active
                        )
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

                        # Check for quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

    except ConnectionError as e:
        print(f"\n[ERROR] Could not connect to Pupil Capture: {e}")
        print("Make sure Pupil Capture is running and Frame Publisher is enabled.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        output.close()
        if pd_sink is not None:
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
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Pupil Capture host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50020,
        help="Pupil Capture control port",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=50,
        help="Size of the gaze region to analyze (pixels)",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=3,
        help="Number of frames to average for smoothing",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video display",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose console output",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction value. Values < 1.0 brighten (e.g., 0.5), "
        "values > 1.0 darken. Default: 1.0 (no correction)",
    )

    # Stability options
    stability_group = parser.add_argument_group("Stability tuning")
    stability_group.add_argument(
        "--note-stability",
        type=int,
        default=2,
        help="Frames for note stability (lower = faster response)",
    )
    stability_group.add_argument(
        "--octave-stability",
        type=int,
        default=3,
        help="Frames for octave stability (higher = more stable)",
    )
    stability_group.add_argument(
        "--octave-threshold",
        type=float,
        default=0.5,
        help="Agreement threshold for octave changes 0-1 (higher = harder to change)",
    )

    # Pure Data output options
    pd_group = parser.add_argument_group("Pure Data output")
    pd_group.add_argument(
        "--pd",
        action="store_true",
        help="Send to Pure Data via FUDI protocol (TCP)",
    )
    pd_group.add_argument(
        "--pd-host",
        type=str,
        default="127.0.0.1",
        help="Pure Data host address",
    )
    pd_group.add_argument(
        "--pd-port",
        type=int,
        default=9001,
        help="Pure Data FUDI port",
    )

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
    )


if __name__ == "__main__":
    main()
