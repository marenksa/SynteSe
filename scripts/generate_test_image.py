#!/usr/bin/env python3
"""Generate a test image grid for color-to-music calibration.

The image is a grid where:
- Rows = color categories (Red, Orange, Yellow, Green, Cyan, Blue, Violet)
- Columns = brightness/octave steps (Octave 2, 3, 4, 5, 6)

Each square's HSV values are chosen to fall within our detection ranges.
"""

import cv2
import numpy as np

# Our hue ranges from analyzer.py (OpenCV uses 0-179)
# Each tuple: (range_start, range_end, center_hue_to_use, color_name, note)
COLOR_DEFINITIONS = [
    (0, 8, 4, "Red", "C"),
    (8, 25, 16, "Orange", "D"),
    (25, 38, 32, "Yellow", "E"),
    (38, 75, 56, "Green", "F"),
    (75, 95, 85, "Cyan", "G"),
    (95, 125, 110, "Blue", "A"),
    (125, 165, 145, "Violet", "B"),
]

# Brightness ranges for each octave
# octave = 2 + int(brightness / 255 * 4)
# Octave 2: 0-63, Octave 3: 64-127, Octave 4: 128-191, Octave 5: 192-254, Octave 6: 255
OCTAVE_DEFINITIONS = [
    (2, 32, "Very Dark"),   # Octave 2: use V=32
    (3, 96, "Dark"),        # Octave 3: use V=96
    (4, 160, "Medium"),     # Octave 4: use V=160
    (5, 210, "Bright"),     # Octave 5: use V=210
    (6, 250, "Very Bright"), # Octave 6: use V=250
]

# Image parameters
SQUARE_SIZE = 100  # pixels per square
LABEL_HEIGHT = 40  # height for row/column labels
PADDING = 5


def create_test_image() -> np.ndarray:
    """Create the color-brightness test grid image."""
    num_colors = len(COLOR_DEFINITIONS)
    num_octaves = len(OCTAVE_DEFINITIONS)

    # Calculate image dimensions
    width = LABEL_HEIGHT + num_octaves * SQUARE_SIZE + PADDING * 2
    height = LABEL_HEIGHT + num_colors * SQUARE_SIZE + PADDING * 2

    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark gray background

    # Draw column headers (octave labels)
    for col, (octave, brightness, label) in enumerate(OCTAVE_DEFINITIONS):
        x = LABEL_HEIGHT + col * SQUARE_SIZE + SQUARE_SIZE // 2
        y = LABEL_HEIGHT - 10
        text = f"Oct {octave}"
        cv2.putText(image, text, (x - 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Also show brightness value
        cv2.putText(image, f"V={brightness}", (x - 25, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Draw row labels and color squares
    for row, (hue_low, hue_high, hue_center, color_name, note) in enumerate(COLOR_DEFINITIONS):
        # Row label
        y_center = LABEL_HEIGHT + row * SQUARE_SIZE + SQUARE_SIZE // 2
        cv2.putText(image, f"{note}", (5, y_center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, color_name, (5, y_center + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Draw squares for each brightness level
        for col, (octave, brightness, _) in enumerate(OCTAVE_DEFINITIONS):
            x1 = LABEL_HEIGHT + col * SQUARE_SIZE + PADDING
            y1 = LABEL_HEIGHT + row * SQUARE_SIZE + PADDING
            x2 = x1 + SQUARE_SIZE - PADDING * 2
            y2 = y1 + SQUARE_SIZE - PADDING * 2

            # Create HSV color and convert to BGR
            hsv_color = np.array([[[hue_center, 255, brightness]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

            # Draw filled rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color.tolist(), -1)

            # Draw border
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # Choose text color based on brightness (dark text on light, light on dark)
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Add hue value in corner
            hue_text = f"H:{hue_center}"
            cv2.putText(image, hue_text, (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    return image


def create_simple_rainbow_strip() -> np.ndarray:
    """Create a simple horizontal rainbow strip for quick testing."""
    strip_height = 80
    strip_width = len(COLOR_DEFINITIONS) * 100

    image = np.ones((strip_height, strip_width, 3), dtype=np.uint8) * 40

    for i, (_, _, hue_center, color_name, note) in enumerate(COLOR_DEFINITIONS):
        x1 = i * 100
        x2 = (i + 1) * 100

        # Use medium brightness (octave 4)
        hsv_color = np.array([[[hue_center, 255, 180]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

        cv2.rectangle(image, (x1, 0), (x2, strip_height), bgr_color.tolist(), -1)
        cv2.putText(image, note, (x1 + 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, color_name, (x1 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main() -> None:
    """Generate and save test images."""
    import os

    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Create test_images directory if it doesn't exist
    output_dir = os.path.join(project_dir, "test_images")
    os.makedirs(output_dir, exist_ok=True)

    # Generate main test grid
    print("Generating color-brightness test grid...")
    grid_image = create_test_image()
    grid_path = os.path.join(output_dir, "color_test_grid.png")
    cv2.imwrite(grid_path, grid_image)
    print(f"  Saved: {grid_path}")

    # Generate simple rainbow strip
    print("Generating rainbow strip...")
    strip_image = create_simple_rainbow_strip()
    strip_path = os.path.join(output_dir, "rainbow_strip.png")
    cv2.imwrite(strip_path, strip_image)
    print(f"  Saved: {strip_path}")

    print()
    print("Test images generated!")
    print()
    print("Color-to-Note mapping (wavelength order):")
    print("  C = Red     (H: 0-8)")
    print("  D = Orange  (H: 8-25)")
    print("  E = Yellow  (H: 25-38)")
    print("  F = Green   (H: 38-75)")
    print("  G = Cyan    (H: 75-95)")
    print("  A = Blue    (H: 95-125)")
    print("  B = Violet  (H: 125-165)")
    print()
    print("Brightness-to-Octave mapping:")
    print("  Octave 2: V = 0-63   (very dark)")
    print("  Octave 3: V = 64-127 (dark)")
    print("  Octave 4: V = 128-191 (medium)")
    print("  Octave 5: V = 192-254 (bright)")
    print("  Octave 6: V = 255    (very bright)")


if __name__ == "__main__":
    main()
