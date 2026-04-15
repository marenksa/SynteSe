"""Main entry point for the Pupil Brightness Tracker."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from pupil_tracker.analyzer import BrightnessAnalyzer, BrightnessReading
from pupil_tracker.client import PupilCaptureClient
from pupil_tracker.output import (
    ConsoleSink,
    ConsoleThresholdSink,
    FileSink,
    MultiSink,
    PureDataFUDISink,
    PureDataSink,
)
from pupil_tracker.processor import FrameProcessor


def draw_brightness_bar(
    frame: np.ndarray,
    brightness: float,
    x: int = 10,
    y: int = 30,
    width: int = 200,
    height: int = 20,
) -> None:
    """Draw a brightness meter on the frame.

    Args:
        frame: The frame to draw on (modified in place).
        brightness: Brightness value (0-255).
        x: X position of the bar.
        y: Y position of the bar.
        width: Width of the bar.
        height: Height of the bar.
    """
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Filled portion
    filled_width = int(brightness / 255 * width)
    if filled_width > 0:
        # Color gradient from dark blue to bright yellow
        ratio = brightness / 255
        color = (
            int(50 + ratio * 50),  # B
            int(50 + ratio * 200),  # G
            int(50 + ratio * 200),  # R
        )
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)

    # Text
    cv2.putText(
        frame,
        f"Brightness: {brightness:.0f}",
        (x + width + 10, y + height - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def run_tracker(
    host: str = "127.0.0.1",
    port: int = 50020,
    region_size: int = 50,
    smoothing: int = 5,
    output_file: str | None = None,
    show_video: bool = True,
    verbose: bool = False,
    pd_osc: bool = False,
    pd_fudi: bool = False,
    pd_host: str = "127.0.0.1",
    pd_port: int | None = None,
) -> None:
    """Run the brightness tracker.

    Args:
        host: Pupil Capture host address.
        port: Pupil Capture control port.
        region_size: Size of the gaze region to analyze.
        smoothing: Number of frames to average for smoothing.
        output_file: Optional file path to log data.
        show_video: Whether to display the video feed.
        verbose: Whether to print verbose console output.
        pd_osc: Enable Pure Data output via OSC (requires mrpeach external).
        pd_fudi: Enable Pure Data output via FUDI (no externals needed).
        pd_host: Pure Data host address.
        pd_port: Pure Data port (default: 9000 for OSC, 9001 for FUDI).
    """
    # Initialize components
    processor = FrameProcessor(region_size=region_size)
    analyzer = BrightnessAnalyzer(smoothing_window=smoothing)

    # Set up output sinks
    output = MultiSink()
    output.add_sink(ConsoleSink(verbose=verbose))
    output.add_sink(ConsoleThresholdSink(low_threshold=50.0, high_threshold=200.0))

    if output_file:
        output.add_sink(FileSink(Path(output_file)))

    # Pure Data outputs
    if pd_osc:
        osc_port = pd_port if pd_port else 9000
        output.add_sink(PureDataSink(host=pd_host, port=osc_port))
    if pd_fudi:
        fudi_port = pd_port if pd_port else 9001
        output.add_sink(PureDataFUDISink(host=pd_host, port=fudi_port))

    print("=" * 60)
    print("Pupil Brightness Tracker")
    print("=" * 60)
    print(f"  Host: {host}:{port}")
    print(f"  Region size: {region_size}px")
    print(f"  Smoothing window: {smoothing} frames")
    print(f"  Video display: {'enabled' if show_video else 'disabled'}")
    if output_file:
        print(f"  Logging to: {output_file}")
    if pd_osc:
        print(f"  Pure Data (OSC): {pd_host}:{pd_port or 9000}")
    if pd_fudi:
        print(f"  Pure Data (FUDI): {pd_host}:{pd_port or 9001}")
    print("=" * 60)
    print("Press 'q' in the video window or Ctrl+C to stop.")
    print()

    last_reading: BrightnessReading | None = None

    try:
        with PupilCaptureClient(host=host, port=port) as client:
            for message in client.stream_realtime():
                # Update gaze (may come with frame or alone)
                if message.gaze is not None:
                    processor.update_gaze(message.gaze)

                # Process on new frame (real-time, no buffering)
                if message.frame is not None:
                    processor.update_frame(message.frame)

                    # Extract gaze region and analyze
                    gaze_region = processor.extract_region()
                    if gaze_region is not None:
                        reading = analyzer.analyze(gaze_region)
                        output.emit(reading)
                        last_reading = reading

                    # Display video with overlay
                    if show_video:
                        frame = processor.get_frame_with_overlay()
                        if frame is not None:
                            if last_reading is not None:
                                draw_brightness_bar(
                                    frame,
                                    last_reading.smoothed_brightness,
                                )
                            cv2.imshow("Pupil Brightness Tracker", frame)

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
        if show_video:
            cv2.destroyAllWindows()

    print("[INFO] Tracker stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stream gaze data from Pupil Core and analyze brightness.",
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
        default=5,
        help="Number of frames to average for brightness smoothing",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for brightness data (JSONL or CSV based on extension)",
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

    # Pure Data output options
    pd_group = parser.add_argument_group("Pure Data output")
    pd_group.add_argument(
        "--pd-osc",
        action="store_true",
        help="Stream to Pure Data via OSC (requires mrpeach external in Pd)",
    )
    pd_group.add_argument(
        "--pd-fudi",
        action="store_true",
        help="Stream to Pure Data via FUDI/TCP (no externals needed)",
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
        default=None,
        help="Pure Data port (default: 9000 for OSC, 9001 for FUDI)",
    )

    args = parser.parse_args()

    run_tracker(
        host=args.host,
        port=args.port,
        region_size=args.region_size,
        smoothing=args.smoothing,
        output_file=args.output,
        show_video=not args.no_video,
        verbose=args.verbose,
        pd_osc=args.pd_osc,
        pd_fudi=args.pd_fudi,
        pd_host=args.pd_host,
        pd_port=args.pd_port,
    )


if __name__ == "__main__":
    main()

