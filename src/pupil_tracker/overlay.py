"""Shared overlay drawing functions for gaze visualization."""

from __future__ import annotations

import cv2
import numpy as np

from pupil_tracker.analyzer import ColorReading, Note


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


def draw_brightness_bar(
    frame: np.ndarray,
    brightness: float,
    x: int = 10,
    y: int = 30,
    width: int = 200,
    height: int = 20,
) -> None:
    """Draw a brightness meter on the frame (modified in place)."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Filled portion
    filled_width = int(brightness / 255 * width)
    if filled_width > 0:
        ratio = brightness / 255
        color = (
            int(50 + ratio * 50),   # B
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


def draw_color_info(
    frame: np.ndarray,
    color_reading: ColorReading,
    x: int = 10,
    y: int = 60,
    size: int = 40,
) -> None:
    """Draw a color square and note name on the frame (modified in place)."""
    note = color_reading.note
    octave = color_reading.octave
    color = NOTE_BGR_COLORS.get(note, (128, 128, 128))
    color_name = NOTE_COLOR_NAMES.get(note, "?")

    # Draw color square with detected color
    cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 200, 200), 2)

    # Draw note name and octave
    note_text = f"{note.name}{octave}"
    cv2.putText(
        frame,
        note_text,
        (x + size + 10, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Draw color name below
    cv2.putText(
        frame,
        color_name,
        (x + size + 10, y + size - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    # Draw MIDI note number
    cv2.putText(
        frame,
        f"MIDI: {color_reading.midi_note}",
        (x + size + 80, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )


def draw_gaze_crosshair(
    frame: np.ndarray,
    x: int,
    y: int,
    confidence: float,
    radius: int = 20,
    confidence_threshold: float = 0.6,
) -> None:
    """Draw a gaze crosshair with confidence indicator on the frame (modified in place)."""
    if confidence >= confidence_threshold:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 165, 255)  # Orange

    # Crosshair circle + lines
    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.line(frame, (x - 30, y), (x + 30, y), color, 1)
    cv2.line(frame, (x, y - 30), (x, y + 30), color, 1)

    # Confidence text
    conf_text = f"Conf: {confidence:.2f}"
    cv2.putText(frame, conf_text, (x + 25, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_region_box(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    region_size: int,
    confidence: float,
    confidence_threshold: float = 0.6,
) -> None:
    """Draw the gaze analysis region box on the frame (modified in place)."""
    if confidence >= confidence_threshold:
        color = (0, 255, 0)
    else:
        color = (0, 165, 255)

    height, width = frame.shape[:2]
    half_size = region_size // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(width, center_x + half_size)
    y2 = min(height, center_y + half_size)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def draw_eye_panel(
    frame: np.ndarray,
    eye_frame: np.ndarray | None,
    *,
    is_blink: bool = False,
    blink_label: str | None = None,
    is_flutter: bool = False,
    flutter_label: str | None = None,
    blink_count: int = 0,
    flutter_count: int = 0,
    margin: int = 10,
    display_width: int = 150,
) -> None:
    """Draw eye camera feed and event indicators in the top-right corner (modified in place).

    Args:
        frame: The world-camera frame to draw on.
        eye_frame: The raw eye camera frame (grayscale or BGR), or None.
        is_blink: Whether a blink is currently active.
        blink_label: Text to show when is_blink is True (e.g. "BLINK 120ms").
        is_flutter: Whether a flutter event is currently active.
        flutter_label: Text to show when is_flutter is True (e.g. "FLUTTER 3 in 2s").
        blink_count: Total blink events detected so far (shown when not active).
        flutter_count: Total flutter events detected so far (shown when not active).
        margin: Pixel margin from frame edges.
        display_width: Target display width for the eye frame.
    """
    height, width = frame.shape[:2]

    if eye_frame is not None:
        # Scale eye frame
        eye_h, eye_w = eye_frame.shape[:2]
        scale = display_width / max(eye_h, eye_w)
        display_size = (int(eye_w * scale), int(eye_h * scale))
        eye_resized = cv2.resize(eye_frame, display_size)
        eye_resized = cv2.flip(eye_resized, 0)  # Pupil eye camera is upside down

        # Convert grayscale to BGR if needed
        if len(eye_resized.shape) == 2:
            eye_resized = cv2.cvtColor(eye_resized, cv2.COLOR_GRAY2BGR)

        eh, ew = eye_resized.shape[:2]
    else:
        # Placeholder when no eye frame available
        ew, eh = display_width, display_width
        eye_resized = np.zeros((eh, ew, 3), dtype=np.uint8)
        cv2.putText(eye_resized, "No eye cam", (10, eh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    x_offset = width - ew - margin
    y_offset = margin

    # Border color based on event state
    if is_blink:
        border_color = (0, 0, 255)  # Red for blink
    else:
        border_color = (200, 200, 200)

    border_width = 3 if (is_blink or is_flutter) else 1
    cv2.rectangle(
        frame,
        (x_offset - border_width, y_offset - border_width),
        (x_offset + ew + border_width, y_offset + eh + border_width),
        border_color,
        border_width,
    )

    # Place eye frame
    frame[y_offset:y_offset + eh, x_offset:x_offset + ew] = eye_resized

    # Blink text
    text_y = y_offset + eh + 20
    if is_blink and blink_label:
        cv2.putText(frame, blink_label, (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Blinks: {blink_count}", (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Flutter text
    text_y += 22
    if is_flutter and flutter_label:
        cv2.putText(frame, flutter_label, (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Flutter: {flutter_count}", (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def build_gamma_lut(gamma: float) -> np.ndarray:
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


def apply_gamma(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply gamma correction to a frame using a precomputed lookup table."""
    return cv2.LUT(frame, lut)
