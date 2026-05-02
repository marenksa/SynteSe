"""Shared overlay drawing functions and per-patch config for gaze visualisation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from eye_synth.signals.env_color import ColorReading


@dataclass
class OverlayConfig:
    """Declares which overlay elements a patch wants displayed.

    Gaze-position elements are on by default; everything else must be
    explicitly requested by each patch via a class-level ``overlay`` attribute.
    """

    # Gaze-position elements — on by default
    show_gaze_crosshair: bool = True
    show_region_box: bool = True

    # Sensor/state readouts — off by default, request per patch
    show_brightness_bar: bool = False
    show_color_info: bool = False
    show_confidence: bool = False
    show_eye_panel: bool = True         # camera feed
    show_blink_flutter: bool = False    # labels/counts below camera; only meaningful with show_eye_panel

    eye_panel_size: int = 150           # display width in px


DEFAULT_OVERLAY = OverlayConfig()


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
    # Derive a display color from the smoothed hue
    hsv_pixel = np.uint8([[[int(color_reading.smoothed_hue), 200, 200]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 200, 200), 2)
    hue_str = f"{color_reading.hue:.0f}" if color_reading.hue is not None else "N/A"
    cv2.putText(frame, f"H:{hue_str}", (x + size + 10, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"S:{color_reading.saturation:.0f}", (x + size + 10, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"V:{color_reading.smoothed_brightness:.0f}", (x + size + 10, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
    show_state: bool = True,
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

    if show_state:
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


def draw_confidence(
    frame: np.ndarray,
    confidence: float,
    x: int = 10,
    y: int = 30,
) -> None:
    color = (0, 255, 0) if confidence >= 0.6 else (0, 165, 255)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def draw_overlay(
    frame: np.ndarray,
    cfg: OverlayConfig,
    *,
    gaze_px: tuple[int, int] | None = None,
    confidence: float = 0.0,
    region_size: int = 50,
    color_reading: ColorReading | None = None,
    eye_frame: np.ndarray | None = None,
    is_blink: bool = False,
    blink_label: str | None = None,
    is_flutter: bool = False,
    flutter_label: str | None = None,
    blink_count: int = 0,
    flutter_count: int = 0,
) -> None:
    """Draw all overlay elements requested by ``cfg`` onto ``frame`` in-place."""
    if cfg.show_gaze_crosshair and gaze_px is not None:
        draw_gaze_crosshair(frame, *gaze_px, confidence)
    if cfg.show_region_box and gaze_px is not None:
        draw_region_box(frame, *gaze_px, region_size, confidence)
    if cfg.show_color_info and color_reading is not None:
        draw_color_info(frame, color_reading, x=10, y=10)
    if cfg.show_brightness_bar and color_reading is not None:
        draw_brightness_bar(frame, color_reading.smoothed_brightness, x=10, y=60)
    if cfg.show_confidence:
        draw_confidence(frame, confidence, x=10, y=30)
    if cfg.show_eye_panel:
        draw_eye_panel(
            frame, eye_frame,
            display_width=cfg.eye_panel_size,
            show_state=cfg.show_blink_flutter,
            is_blink=is_blink, blink_label=blink_label,
            is_flutter=is_flutter, flutter_label=flutter_label,
            blink_count=blink_count, flutter_count=flutter_count,
        )


