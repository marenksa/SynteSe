"""Video player with gaze overlay for Pupil Capture recordings."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pupil_tracker.recording import Recording


class GazeVideoPlayer:
    """Interactive video player that displays gaze position overlay."""

    def __init__(self, recording: Recording):
        self.recording = recording
        self.frame_index = 0
        self.playing = False
        self.playback_speed = 1.0

        # Display settings
        self.gaze_radius = 20
        self.gaze_color = (0, 255, 0)  # Green
        self.gaze_thickness = 2
        self.low_confidence_color = (0, 165, 255)  # Orange
        self.confidence_threshold = 0.6

        # Window name
        self.window_name = f"Gaze Player - {recording.recording_name}"

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

            # Draw overlays
            display_frame = self.draw_gaze(current_frame.copy(), self.frame_index)
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

    args = parser.parse_args()

    # Validate path
    recording_path = Path(args.recording_path)
    if not recording_path.exists():
        print(f"Error: Recording not found: {recording_path}")
        sys.exit(1)

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

            player = GazeVideoPlayer(recording)
            player.frame_index = args.start_frame
            player.playing = args.autoplay
            player.run()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
