"""
Real-Time Spectrogram (Module 3).

This module renders a scrolling spectrogram on a pygame Surface, showing
frequency content over time. It uses numpy's FFT (via the AudioEngine's
get_frequency_data method) to compute the spectrum, then maps energy levels
to colors using a dark-blue → cyan → green → yellow → red → white color map.

Features:
  - Scrolling display: new FFT frames appear on the right, old frames scroll left
  - Frequency axis labeled in kHz (0 to Nyquist)
  - Color map: dark blue (low energy) → white (high energy)
  - Configurable FFT size (512, 1024, 2048, 4096)
  - Hann window applied to each FFT frame (done in AudioEngine)
  - Dual-panel support: shows both original and shifted spectrograms stacked
    vertically when playing time-expanded audio
  - Time grid lines every 1 second
"""

import numpy as np
import pygame
from collections import deque


# ---------------------------------------------------------------------------
# Color map: energy value (0–255) → (r, g, b)
# ---------------------------------------------------------------------------

def energy_to_color(value: float) -> tuple[int, int, int]:
    """
    Map an energy value (0–255) to an RGB color.

    Color progression (matching the original HTML app):
      0.0–0.2:  dark blue    (low energy)
      0.2–0.4:  cyan
      0.4–0.6:  green
      0.6–0.8:  yellow
      0.8–1.0:  red
      1.0+:     white        (peak energy)

    This progression uses brightness differences in addition to color,
    making it partially color-blind safe.

    Args:
        value: Energy value from FFT magnitude, scaled 0–255.

    Returns:
        (r, g, b) tuple, each 0–255.
    """
    normalized = max(0.0, min(1.0, value / 255.0))

    if normalized < 0.2:
        # Dark blue — near-silence
        # Interpolate from (0,0,20) to (0,0,80)
        t = normalized / 0.2
        return (0, 0, int(20 + t * 60))
    elif normalized < 0.4:
        # Blue → cyan
        t = (normalized - 0.2) / 0.2
        return (0, int(204 * t), int(80 + (204 - 80) * t))
    elif normalized < 0.6:
        # Cyan → green
        t = (normalized - 0.4) / 0.2
        return (0, 204, int(204 - (204 - 100) * t))
    elif normalized < 0.8:
        # Green → yellow
        t = (normalized - 0.6) / 0.2
        return (int(255 * t), 255, 0)
    elif normalized < 1.0:
        # Yellow → red
        t = (normalized - 0.8) / 0.2
        return (255, int(255 - 255 * t), 0)
    else:
        # White — peak
        return (255, 255, 255)


# ---------------------------------------------------------------------------
# Spectrogram Renderer
# ---------------------------------------------------------------------------

class Spectrogram:
    """
    A scrolling spectrogram rendered on a pygame Surface.

    The spectrogram maintains a history of FFT frames in a deque. Each frame
    is a 1D array of energy values (0–255). The renderer draws these frames
    as vertical columns, with the newest frame on the right and older frames
    scrolling left.

    Attributes:
        width:       Pixel width of the spectrogram panel.
        height:      Pixel height of the spectrogram panel.
        num_bins:    Number of frequency bins (determined by FFT size).
        history:     Deque of recent FFT frames (each is a numpy array).
        max_history: Maximum number of frames to keep (width of the panel).
    """

    def __init__(self, width: int = 600, height: int = 200, max_history: int = 600):
        """
        Initialize the spectrogram renderer.

        Args:
            width:       Pixel width of the spectrogram panel.
            height:      Pixel height of the spectrogram panel.
            max_history: Maximum number of FFT frames to store (one per pixel column).
        """
        self.width = width
        self.height = height
        self.max_history = max_history
        self.history: deque[np.ndarray] = deque(maxlen=max_history)

        # Pre-create the off-screen surface for rendering.
        self._surface = pygame.Surface((width, height))

        # For dual-panel mode (original + shifted), we create a second surface.
        self._original_surface = pygame.Surface((width, height))

        # Track time for grid lines.
        self._frame_count = 0
        self._sample_rate = 44100

    def update(self, freq_data: np.ndarray, sample_rate: int) -> None:
        """
        Add a new FFT frame to the spectrogram history.

        Called once per frame from the main loop. The new frame is appended
        to the history deque; old frames are automatically dropped when the
        deque is full.

        Args:
            freq_data:   FFT magnitude data (0–255 scaled) from AudioEngine.
            sample_rate: Current sample rate (for frequency axis labeling).
        """
        if freq_data is not None and len(freq_data) > 0:
            self.history.append(freq_data.copy())
        self._sample_rate = sample_rate
        self._frame_count += 1

    def render(self, sample_rate: int, show_original: bool = False,
               original_data: np.ndarray | None = None) -> pygame.Surface:
        """
        Render the spectrogram to a pygame Surface.

        Each column of the surface represents one FFT frame in history.
        The vertical axis represents frequency bins, mapped from
        (top = Nyquist) to (bottom = 0 Hz).

        Args:
            sample_rate:   Current sample rate for frequency axis labels.
            show_original: If True, render a dual-panel (original above, shifted below).
            original_data: FFT data for the original (unshifted) audio, for the top panel.

        Returns:
            A pygame Surface containing the rendered spectrogram.
        """
        if show_original:
            return self._render_dual(sample_rate, original_data)
        else:
            return self._render_single(sample_rate)

    def _render_single(self, sample_rate: int) -> pygame.Surface:
        """Render a single-panel spectrogram."""
        surf = self._surface
        surf.fill((10, 10, 26))  # Dark background

        num_frames = len(self.history)
        if num_frames == 0:
            self._draw_frequency_axis(surf, sample_rate)
            return surf

        # Get the most recent frame to determine bin count.
        last_frame = self.history[-1]
        num_bins = len(last_frame)

        # Draw each frame as a vertical column.
        for col_idx, frame in enumerate(self.history):
            x = col_idx  # One pixel per frame
            if x >= self.width:
                break

            for bin_idx in range(num_bins):
                # Map bin index to y coordinate.
                # Higher frequencies at top, lower at bottom.
                y = self.height - 1 - int((bin_idx / max(1, num_bins - 1)) * (self.height - 1))
                if 0 <= y < self.height:
                    color = energy_to_color(float(frame[bin_idx]))
                    surf.set_at((x, y), color)

        self._draw_frequency_axis(surf, sample_rate)
        self._draw_time_grid(surf, sample_rate)
        return surf

    def _render_dual(self, sample_rate: int, original_data: np.ndarray | None) -> pygame.Surface:
        """
        Render a dual-panel spectrogram (original above, shifted below).

        The top panel shows the original ultrasonic frequencies.
        The bottom panel shows the shifted (audible) frequencies.
        """
        # Top panel: original spectrum
        self._original_surface.fill((10, 10, 26))
        if original_data is not None and len(original_data) > 0:
            num_bins = len(original_data)
            for bin_idx in range(num_bins):
                y = self.height - 1 - int((bin_idx / max(1, num_bins - 1)) * (self.height - 1))
                if 0 <= y < self.height:
                    color = energy_to_color(float(original_data[bin_idx]))
                    # Draw as a vertical bar (full width for a static snapshot).
                    for x in range(self.width):
                        self._original_surface.set_at((x, y), color)

        self._draw_frequency_axis(self._original_surface, sample_rate, label="ORIGINAL")

        # Bottom panel: shifted spectrum (from history)
        bottom = self._render_single(sample_rate)

        # Combine into a single surface (stacked vertically).
        combined = pygame.Surface((self.width, self.height * 2 + 10))
        combined.fill((10, 10, 26))
        combined.blit(self._original_surface, (0, 0))
        combined.blit(bottom, (0, self.height + 10))
        return combined

    def _draw_frequency_axis(self, surf: pygame.Surface, sample_rate: int,
                             label: str = "") -> None:
        """
        Draw frequency axis labels on the left side of the spectrogram.

        Labels are in kHz, from 0 to Nyquist (sample_rate / 2).
        """
        font = pygame.font.SysFont("monospace", 10)
        nyquist = sample_rate / 2.0
        max_khz = nyquist / 1000.0

        # Choose label interval to show ~6 labels.
        if max_khz > 80:
            step = 20
        elif max_khz > 40:
            step = 10
        elif max_khz > 20:
            step = 5
        else:
            step = 2

        for f_khz in range(0, int(max_khz) + 1, step):
            y = self.height - 1 - int((f_khz * 1000 / max(1.0, nyquist)) * (self.height - 1))
            if 0 <= y < self.height:
                text = font.render(f"{f_khz}k", True, (224, 224, 255))
                surf.blit(text, (2, y - 5))

        # Optional panel label (e.g. "ORIGINAL" or "SHIFTED").
        if label:
            text = font.render(label, True, (0, 204, 204))
            surf.blit(text, (self.width - 60, 5))

    def _draw_time_grid(self, surf: pygame.Surface, sample_rate: int) -> None:
        """
        Draw vertical time grid lines every 1 second.

        Based on the frame count and an estimated frame rate (~60 FPS).
        """
        # Estimate: at 60 FPS, one second = 60 frames.
        frames_per_second = 60
        frames_per_second_mark = frames_per_second

        for sec in range(1, 10):
            x = self.width - 1 - (sec * frames_per_second_mark)
            if x < 0:
                break
            pygame.draw.line(surf, (40, 40, 60), (x, 0), (x, self.height), 1)

    def clear(self) -> None:
        """Clear the spectrogram history."""
        self.history.clear()
        self._frame_count = 0