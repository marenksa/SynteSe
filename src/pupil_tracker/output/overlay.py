"""Shared overlay drawing functions for gaze visualisation."""

from __future__ import annotations

import cv2
import numpy as np

from pupil_tracker.signals.env_color import ColorReading, Note


NOTE_BGR_COLORS: dict[Note, tuple[int, int, int]] = {
    Note.C: (60, 60, 220),
    Note.D: (60, 140, 255),
    Note.E: (60, 220, 255),
    Note.F: (60, 180, 60),
    Note.G: (180, 180, 60),
    Note.A: (220, 120, 60),
    Note.B: (180, 60, 180),
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
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    filled_width = int(brightness / 255 * width)
    if filled_width > 0:
        ratio = brightness / 255
        color = (int(50 + ratio * 50), int(50 + ratio * 200), int(50 + ratio * 200))
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    cv2.putText(frame, f"Brightness: {brightness:.0f}",
                (x + width + 10, y + height - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_color_info(
    frame: np.ndarray,
    color_reading: ColorReading,
    x: int = 10,
    y: int = 60,
    size: int = 40,
) -> None:
    note = color_reading.note
    octave = color_reading.octave
    color = NOTE_BGR_COLORS.get(note, (128, 128, 128))
    color_name = NOTE_COLOR_NAMES.get(note, "?")

    cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 200, 200), 2)
    cv2.putText(frame, f"{note.name}{octave}", (x + size + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, color_name, (x + size + 10, y + size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"MIDI: {color_reading.midi_note}", (x + size + 80, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def draw_gaze_crosshair(
    frame: np.ndarray,
    x: int,
    y: int,
    confidence: float,
    radius: int = 20,
    confidence_threshold: float = 0.6,
) -> None:
    color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)
    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.line(frame, (x - 30, y), (x + 30, y), color, 1)
    cv2.line(frame, (x, y - 30), (x, y + 30), color, 1)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (x + 25, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_region_box(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    region_size: int,
    confidence: float,
    confidence_threshold: float = 0.6,
) -> None:
    color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)
    height, width = frame.shape[:2]
    half = region_size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(width, center_x + half)
    y2 = min(height, center_y + half)
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
    height, width = frame.shape[:2]

    if eye_frame is not None:
        eye_h, eye_w = eye_frame.shape[:2]
        scale = display_width / max(eye_h, eye_w)
        eye_resized = cv2.resize(eye_frame, (int(eye_w * scale), int(eye_h * scale)))
        eye_resized = cv2.flip(eye_resized, 0)
        if len(eye_resized.shape) == 2:
            eye_resized = cv2.cvtColor(eye_resized, cv2.COLOR_GRAY2BGR)
        eh, ew = eye_resized.shape[:2]
    else:
        ew, eh = display_width, display_width
        eye_resized = np.zeros((eh, ew, 3), dtype=np.uint8)
        cv2.putText(eye_resized, "No eye cam", (10, eh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    x_offset = width - ew - margin
    y_offset = margin

    border_color = (0, 0, 255) if is_blink else (200, 200, 200)
    border_width = 3 if (is_blink or is_flutter) else 1
    cv2.rectangle(frame,
                  (x_offset - border_width, y_offset - border_width),
                  (x_offset + ew + border_width, y_offset + eh + border_width),
                  border_color, border_width)
    frame[y_offset:y_offset + eh, x_offset:x_offset + ew] = eye_resized

    text_y = y_offset + eh + 20
    if is_blink and blink_label:
        cv2.putText(frame, blink_label, (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Blinks: {blink_count}", (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    text_y += 22
    if is_flutter and flutter_label:
        cv2.putText(frame, flutter_label, (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Flutter: {flutter_count}", (x_offset, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def build_gamma_lut(gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return np.arange(256, dtype=np.uint8)
    return np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )


def apply_gamma(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(frame, lut)
