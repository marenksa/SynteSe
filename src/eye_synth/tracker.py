"""Live tracker entry point."""

import argparse
import sys
import time

import cv2
import numpy as np

from eye_synth.input.live import PupilCaptureClient
from eye_synth.output import (
    ColorConsoleSink, DEFAULT_OVERLAY, MultiSink, PureDataSink,
    apply_gamma, build_gamma_lut, draw_overlay,
)
from eye_synth.patches import load_patch
from eye_synth.signals.bus import OutputBus
from eye_synth.signals.pipeline import Pipeline


def run_tracker(
    host: str = "127.0.0.1",
    port: int = 50020,
    verbose: bool = False,
    pd_host: str = "127.0.0.1",
    pd_port: int = 9001,
    gamma: float = 1.0,
    patch_name: str = "TNC_v1",
    show_overlay: bool = True,
) -> None:
    """Run the live tracker."""
    pipeline = Pipeline()
    gamma_lut = build_gamma_lut(gamma)

    pd_sink = PureDataSink(host=pd_host, port=pd_port)
    outputs = OutputBus(pd_sink)
    patch = load_patch(patch_name)
    overlay_cfg = getattr(patch, 'overlay', DEFAULT_OVERLAY)

    console_output = MultiSink()
    console_output.add_sink(ColorConsoleSink(verbose=verbose))

    # Display state (overlay only, not sent to outputs)
    blink_flash_until = 0.0
    last_blink_label: str | None = None
    flutter_flash_until = 0.0
    last_flutter_label: str | None = None
    latest_eye_frame: np.ndarray | None = None

    print("=" * 60)
    print("Pupil Tracker")
    print("=" * 60)
    print(f"  Host: {host}:{port}")
    print(f"  Patch: {patch_name}")
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

                # Apply gamma before passing to pipeline and display
                frame_data = None
                if message.frame is not None and gamma != 1.0:
                    frame_data = apply_gamma(message.frame.data, gamma_lut)

                signals = pipeline.process_live(message, now, frame_data=frame_data)

                # Update display flash state from signals
                if signals.eye.blink is not None:
                    blink_flash_until = now + 0.3
                    b = signals.eye.blink
                    last_blink_label = (
                        f"{b.blink_type.value.upper()} {b.duration_ms:.0f}ms"
                        if b.duration_ms >= 0 else "BLINK"
                    )
                if pipeline.blink_tracker.is_flutter_active:
                    last_flutter_label = (
                        f"FLUTTER {pipeline.blink_tracker.active_flutter_blink_count} blinks"
                    )
                if signals.eye.flutter is not None:
                    flutter_flash_until = now + 0.3
                    last_flutter_label = f"FLUTTER {signals.eye.flutter.blink_count} blinks"

                if message.eye_frame is not None:
                    latest_eye_frame = message.eye_frame.data

                if signals.has_env_reading and pipeline.last_color_reading is not None:
                    console_output.emit(pipeline.last_color_reading)

                # Draw overlay
                if message.frame is not None:
                    display = (
                        frame_data if frame_data is not None else message.frame.data
                    ).copy()

                    if show_overlay:
                        gaze_px = None
                        confidence = 0.0
                        if pipeline.processor.last_gaze is not None:
                            gx, gy = pipeline.processor.norm_to_pixel(
                                pipeline.processor.last_gaze.norm_pos[0],
                                pipeline.processor.last_gaze.norm_pos[1],
                                message.frame.width,
                                message.frame.height,
                            )
                            gaze_px = (gx, gy)
                            confidence = pipeline.processor.last_raw_confidence

                        is_blink = now < blink_flash_until
                        is_flutter = now < flutter_flash_until or pipeline.blink_tracker.is_flutter_active
                        draw_overlay(
                            display, overlay_cfg,
                            gaze_px=gaze_px,
                            confidence=confidence,
                            region_size=pipeline.processor.region_size,
                            color_reading=pipeline.last_color_reading,
                            eye_frame=latest_eye_frame,
                            is_blink=is_blink,
                            blink_label=last_blink_label if is_blink else None,
                            is_flutter=is_flutter,
                            flutter_label=last_flutter_label if is_flutter else None,
                            blink_count=pipeline.blink_tracker.blink_count,
                            flutter_count=pipeline.blink_tracker.flutter_count,
                        )

                    cv2.imshow("Pupil Tracker", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                patch.update(signals, outputs)

    except ConnectionError as e:
        print(f"\n[ERROR] Could not connect to Pupil Capture: {e}")
        print("Make sure Pupil Capture is running and Frame Publisher is enabled.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        console_output.close()
        patch.shutdown(outputs)
        pd_sink.close()
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
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (< 1.0 brightens, > 1.0 darkens)")
    parser.add_argument("--patch", type=str, default="TNC_v1",
                        help="Patch to use for mapping signals to outputs")

    parser.add_argument("--pd-host", type=str, default="127.0.0.1",
                        help="Pure Data host (default: 127.0.0.1)")
    parser.add_argument("--pd-port", type=int, default=9001,
                        help="Pure Data port (default: 9001)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable colour/brightness overlay")

    args = parser.parse_args()

    run_tracker(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        pd_host=args.pd_host,
        pd_port=args.pd_port,
        gamma=args.gamma,
        patch_name=args.patch,
        show_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
