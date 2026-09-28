"""
Canvas UI & Layout (Module 7).

This module implements the entire user interface using pygame's canvas drawing
primitives — no DOM elements, no HTML widgets, just pixels on a Surface.
It mirrors the original HTML app's "all UI is canvas-drawn" philosophy.

Layout (top to bottom, 800×900 pixels):
  ┌──────────────────────────────────────────────┐
  │ Title Bar (50px)                             │
  ├──────────────────────────────────────────────┤
  │ Transport Bar (40px) — Play/Pause/Stop, File │
  │                          Load, Live Mic, Prog│
  ├──────────────────────────────────────────────┤
  │ Method Selector (40px) — 3 buttons           │
  ├──────────────────────────────────────────────┤
  │ Method Controls (60px) — sliders/buttons     │
  ├──────────────────────────────────────────────┤
  │ Spectrogram Panel (200px×600px)              │
  ├──────────────────────────────────────────────┤
  │ Species Panel (180px) — reference table      │
  ├──────────────────────────────────────────────┤
  │ Filter Panel (120px) — toggles + sliders     │
  ├──────────────────────────────────────────────┤
  │ Record/Export Bar (40px)                     │
  ├──────────────────────────────────────────────┤
  │ Status Bar (30px)                            │
  └──────────────────────────────────────────────┘

The module provides:
  - CanvasButton: clickable rectangle with hover/active states + text label
  - CanvasSlider: draggable horizontal slider with label + value display
  - CanvasToggle: on/off checkbox-style button
  - UICanvas: the main UI manager that lays out and renders all components

Mouse handling: the UICanvas tracks mouse position and click events, detects
which UI region was clicked, and dispatches to the appropriate handler.

Keyboard shortcuts:
  Space = play/pause, R = record toggle, 1/2/3 = select method,
  S = stop, E = export spectrogram, L = load file, M = live mic
"""

import pygame
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Color Scheme (matching the original HTML app)
# ---------------------------------------------------------------------------
COLORS = {
    "bg":          (10, 10, 26),       # #0a0a1a — dark navy background
    "panel_bg":    (26, 26, 51),       # #1a1a33 — panel background
    "accent":      (0, 204, 204),      # #00cccc — cyan/teal
    "active":      (0, 255, 127),      # #00ff7f — green for active states
    "warning":     (255, 170, 0),      # #ffaa00 — orange for warnings
    "error":       (255, 68, 68),      # #ff4444 — red for recording/errors
    "text":        (224, 224, 255),    # #e0e0ff — light text
    "text_dim":    (128, 128, 160),    # dimmed text
    "slider_bg":   (51, 51, 68),       # #333 — slider track
    "slider_fill": (0, 180, 180),      # lighter cyan for slider fill
    "border":      (0, 204, 204),      # panel borders
}

# Canvas dimensions
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 900

# Layout regions (y-coordinates of each section)
LAYOUT = {
    "title":     {"y": 0,   "h": 50},
    "transport": {"y": 50,  "h": 50},
    "method":    {"y": 100, "h": 40},
    "controls":  {"y": 140, "h": 60},
    "spectrogram":{"y": 200, "h": 200},
    "species":   {"y": 400, "h": 180},
    "filters":   {"y": 580, "h": 120},
    "record":    {"y": 700, "h": 40},
    "status":    {"y": 740, "h": 30},
}

# Valid FFT sizes
FFT_SIZES = [512, 1024, 2048, 4096]

# Valid frequency division divisors
DIVISORS = [2, 4, 8, 10, 16]


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

@dataclass
class CanvasButton:
    """
    A clickable rectangular button drawn on the canvas.

    Attributes:
        x, y:       Top-left pixel position.
        w, h:       Width and height in pixels.
        label:      Text displayed on the button.
        on_click:   Callback function called when the button is clicked.
        active:     Whether this button is currently selected/active
                    (changes its color to green).
        hover:      Whether the mouse is currently over this button.
        pressed:    Whether the mouse button is held down over this button.
        color:      Base color (defaults to accent cyan).
    """
    x: int
    y: int
    w: int
    h: int
    label: str
    on_click: Callable = None
    active: bool = False
    hover: bool = False
    pressed: bool = False
    color: tuple = None

    def contains(self, mx: int, my: int) -> bool:
        """Check if a point (mx, my) is inside this button."""
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Render this button on the given surface.

        Button appearance changes based on state:
          - Active (selected): green background
          - Hover (mouse over): brighter border
          - Pressed: slightly inset (darker background)
          - Normal: accent color border, panel background
        """
        # Determine background color based on state.
        if self.active:
            bg = COLORS["active"]
            fg = COLORS["bg"]
        elif self.pressed:
            bg = (20, 20, 40)
            fg = COLORS["text"]
        else:
            bg = COLORS["panel_bg"]
            fg = COLORS["text"]

        # Determine border color.
        border = COLORS["warning"] if self.hover else (self.color or COLORS["accent"])

        # Draw button background and border.
        pygame.draw.rect(surf, bg, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, border, (self.x, self.y, self.w, self.h), 2)

        # Draw centered label text.
        text_surf = font.render(self.label, True, fg)
        text_rect = text_surf.get_rect(center=(self.x + self.w // 2, self.y + self.h // 2))
        surf.blit(text_surf, text_rect)


@dataclass
class CanvasSlider:
    """
    A horizontal slider drawn on the canvas.

    Attributes:
        x, y:       Top-left pixel position of the slider track.
        w, h:       Width and height of the track.
        min_val:    Minimum value.
        max_val:    Maximum value.
        value:      Current value.
        label:      Text label displayed above the slider.
        on_change:  Callback called with the new value when the slider moves.
        dragging:   Whether the user is currently dragging the thumb.
        integer:    If True, snap value to the nearest integer.
    """
    x: int
    y: int
    w: int
    h: int
    min_val: float
    max_val: float
    value: float
    label: str
    on_change: Callable = None
    dragging: bool = False
    integer: bool = False

    def contains(self, mx: int, my: int) -> bool:
        """Check if a point is inside the slider's clickable area (track + thumb)."""
        # Expand the hit area vertically for easier interaction.
        return self.x - 5 <= mx <= self.x + self.w + 5 and self.y - 10 <= my <= self.y + self.h + 10

    def update_value_from_mouse(self, mx: int) -> None:
        """Update the slider value based on the mouse x position."""
        # Normalize mouse x to 0–1 range.
        normalized = max(0.0, min(1.0, (mx - self.x) / self.w))
        new_val = self.min_val + normalized * (self.max_val - self.min_val)
        if self.integer:
            new_val = round(new_val)
        self.value = new_val
        if self.on_change:
            self.on_change(new_val)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        """
        Render the slider on the given surface.

        Draws:
          - Label text above the slider
          - Track (dark rectangle)
          - Fill (colored portion up to the current value)
          - Thumb (circle at the current value position)
          - Value text below the thumb
        """
        # Draw label above slider.
        label_surf = small_font.render(self.label, True, COLORS["text"])
        surf.blit(label_surf, (self.x, self.y - 16))

        # Draw track background.
        pygame.draw.rect(surf, COLORS["slider_bg"], (self.x, self.y, self.w, self.h))

        # Draw fill (colored portion up to the current value).
        normalized = (self.value - self.min_val) / max(0.001, self.max_val - self.min_val)
        fill_w = int(normalized * self.w)
        pygame.draw.rect(surf, COLORS["slider_fill"], (self.x, self.y, fill_w, self.h))

        # Draw thumb (circle at the current position).
        thumb_x = self.x + fill_w
        thumb_y = self.y + self.h // 2
        thumb_color = COLORS["warning"] if self.dragging else COLORS["accent"]
        pygame.draw.circle(surf, thumb_color, (thumb_x, thumb_y), 10)
        pygame.draw.circle(surf, COLORS["bg"], (thumb_x, thumb_y), 10, 2)

        # Draw value text.
        if self.integer:
            val_text = f"{int(self.value)}"
        else:
            val_text = f"{self.value:.1f}"
        val_surf = small_font.render(val_text, True, COLORS["text"])
        surf.blit(val_surf, (thumb_x - 15, self.y + self.h + 4))


@dataclass
class CanvasToggle:
    """
    A toggle button (checkbox style) drawn on the canvas.

    Attributes:
        x, y:    Top-left position of the toggle.
        w, h:    Size of the toggle box.
        label:   Text label next to the toggle.
        active:  Whether the toggle is on or off.
        on_toggle: Callback called when the toggle state changes.
    """
    x: int
    y: int
    w: int
    h: int
    label: str
    active: bool = False
    on_toggle: Callable = None

    def contains(self, mx: int, my: int) -> bool:
        """Check if a point is inside the toggle's clickable area."""
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def toggle(self) -> None:
        """Flip the toggle state and call the callback."""
        self.active = not self.active
        if self.on_toggle:
            self.on_toggle(self.active)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        """
        Render the toggle on the given surface.

        Appearance:
          - Active: green box with a checkmark
          - Inactive: dark box with empty interior
        """
        # Draw box.
        color = COLORS["active"] if self.active else COLORS["slider_bg"]
        pygame.draw.rect(surf, color, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, COLORS["accent"], (self.x, self.y, self.w, self.h), 1)

        # Draw checkmark if active.
        if self.active:
            cx, cy = self.x + self.w // 2, self.y + self.h // 2
            pygame.draw.line(surf, COLORS["bg"], (cx - 4, cy), (cx - 1, cy + 3), 2)
            pygame.draw.line(surf, COLORS["bg"], (cx - 1, cy + 3), (cx + 4, cy - 3), 2)

        # Draw label.
        label_surf = font.render(self.label, True, COLORS["text"])
        surf.blit(label_surf, (self.x + self.w + 8, self.y))


# ---------------------------------------------------------------------------
# Main UI Canvas Manager
# ---------------------------------------------------------------------------

class UICanvas:
    """
    The main UI manager. Handles layout, rendering, and event dispatch
    for all canvas-drawn UI components.

    This class creates and manages all buttons, sliders, toggles, and panels,
    and provides methods for:
      - handle_mouse_down(mx, my): process a mouse click
      - handle_mouse_motion(mx, my): update hover/dragging states
      - handle_mouse_up(mx, my): release mouse button
      - draw(screen, app_state): render the full UI on the screen
    """

    def __init__(self):
        """Initialize all UI components with their positions and callbacks."""
        # Fonts (initialized lazily — pygame.font.init must be called first).
        self._font: pygame.font.Font = None
        self._small_font: pygame.font.Font = None
        self._tiny_font: pygame.font.Font = None
        self._title_font: pygame.font.Font = None

        # --- Transport Buttons ---
        btn_w, btn_h = 70, 30
        transport_y = LAYOUT["transport"]["y"] + 10
        self.btn_play = CanvasButton(50, transport_y, btn_w, btn_h, "▶ Play")
        self.btn_pause = CanvasButton(130, transport_y, btn_w, btn_h, "❚❚ Pause")
        self.btn_stop = CanvasButton(210, transport_y, btn_w, btn_h, "■ Stop")
        self.btn_load = CanvasButton(300, transport_y, btn_w, btn_h, "📁 Load WAV")
        self.btn_mic = CanvasButton(380, transport_y, btn_w, btn_h, "🎙 Live Mic")

        # --- Method Selector Buttons ---
        method_y = LAYOUT["method"]["y"]
        method_w = 200
        self.btn_time_exp = CanvasButton(50, method_y, method_w, 30, "Time Expansion")
        self.btn_heterodyne = CanvasButton(270, method_y, method_w, 30, "Heterodyne")
        self.btn_freq_div = CanvasButton(490, method_y, method_w, 30, "Freq Division")

        # --- Method Control Slider ---
        # Used for expansion factor (1-20), oscillator freq (20-120 kHz), or divisor.
        # The slider's range and label update based on the selected method.
        self.slider_method = CanvasSlider(
            x=50, y=LAYOUT["controls"]["y"] + 20,
            w=400, h=12,
            min_val=1, max_val=20, value=10,
            label="Expansion Factor (1x–20x)",
        )

        # --- FFT Size Buttons ---
        fft_y = LAYOUT["controls"]["y"] + 20
        self.fft_buttons: list[CanvasButton] = []
        for i, size in enumerate(FFT_SIZES):
            btn = CanvasButton(500 + i * 70, fft_y, 60, 25, str(size))
            btn.active = (size == 2048)
            self.fft_buttons.append(btn)

        # --- Filter Toggles ---
        filter_y = LAYOUT["filters"]["y"]
        self.filter_toggles: dict[str, CanvasToggle] = {}
        self.filter_sliders: dict[str, CanvasSlider] = {}

        filter_defs = [
            ("highpass",  "HP Filter",    "cutoff",   1,     20000),
            ("lowpass",   "LP Filter",    "cutoff",   2,     96000),
            ("bandpass",  "BP Filter",    "center",   20,    120000),
            ("noise_gate","Noise Gate",   "threshold",-80,   -20),
            ("gain",      "Volume Boost", "gain",     1,     10),
        ]

        for i, (name, label, param, min_v, max_v) in enumerate(filter_defs):
            row_y = filter_y + 10 + i * 22
            # Toggle on the left.
            toggle = CanvasToggle(55, row_y, 16, 16, f"{label}:", active=False)
            self.filter_toggles[name] = toggle

            # Slider on the right (showing the parameter value).
            slider = CanvasSlider(
                x=200, y=row_y + 2, w=200, h=8,
                min_val=min_v, max_val=max_v, value=(min_v + max_v) / 2,
                label=f"{param.title()}", integer=False,
            )
            # Special: noise gate threshold should be integer.
            if name == "noise_gate":
                slider.integer = True
            self.filter_sliders[name] = slider

        # --- Record / Export Buttons ---
        record_y = LAYOUT["record"]["y"]
        self.btn_record = CanvasButton(50, record_y, 100, 30, "● RECORD")
        self.btn_export_wav = CanvasButton(170, record_y, 110, 30, "Export WAV")
        self.btn_export_png = CanvasButton(300, record_y, 110, 30, "Export PNG")

        # --- State tracking ---
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_down = False

        # Callbacks — set by main.py to connect UI to the audio engine.
        # Using a callback dict so main.py can wire everything up.
        self.callbacks: dict[str, Callable] = {}

        # Currently selected species (for highlight in the species panel).
        self._selected_species_idx: int | None = None
        self._species_scroll = 0  # scroll offset for the species panel

        # Progress bar state (updated from main loop).
        self._progress = 0.0  # 0.0 to 1.0
        self._time_text = "0.0s / 0.0s"

        # File info text.
        self._file_info = "No file loaded"

        # Match indicator.
        self._match_text = ""
        self._match_species_idx: int | None = None

    def init_fonts(self) -> None:
        """Initialize fonts. Must be called after pygame.font.init()."""
        self._font = pygame.font.SysFont("monospace", 14)
        self._small_font = pygame.font.SysFont("monospace", 12)
        self._tiny_font = pygame.font.SysFont("monospace", 10)
        self._title_font = pygame.font.SysFont("monospace", 22, bold=True)

    def set_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function for a UI action."""
        self.callbacks[name] = callback

    # -----------------------------------------------------------------------
    # Event Handling
    # -----------------------------------------------------------------------

    def handle_mouse_down(self, mx: int, my: int) -> None:
        """Process a mouse-down event (click)."""
        self._mouse_x = mx
        self._mouse_y = my
        self._mouse_down = True

        # Check transport buttons.
        for btn in [self.btn_play, self.btn_pause, self.btn_stop,
                    self.btn_load, self.btn_mic]:
            if btn.contains(mx, my):
                btn.pressed = True
                if btn.on_click:
                    btn.on_click()
                return

        # Check method selector buttons.
        for btn, method_key in [(self.btn_time_exp, "time_expansion"),
                                 (self.btn_heterodyne, "heterodyne"),
                                 (self.btn_freq_div, "frequency_division")]:
            if btn.contains(mx, my):
                self._select_method(method_key)
                return

        # Check FFT size buttons.
        for btn in self.fft_buttons:
            if btn.contains(mx, my):
                for b in self.fft_buttons:
                    b.active = False
                btn.active = True
                if self.callbacks.get("fft_change"):
                    self.callbacks["fft_change"](int(btn.label))
                return

        # Check method control slider.
        if self.slider_method.contains(mx, my):
            self.slider_method.dragging = True
            self.slider_method.update_value_from_mouse(mx)
            if self.callbacks.get("method_value_change"):
                self.callbacks["method_value_change"](self.slider_method.value)
            return

        # Check filter toggles and sliders.
        for name, toggle in self.filter_toggles.items():
            if toggle.contains(mx, my):
                toggle.toggle()
                if self.callbacks.get("filter_toggle"):
                    self.callbacks["filter_toggle"](name, toggle.active)
                return

        for name, slider in self.filter_sliders.items():
            if slider.contains(mx, my):
                slider.dragging = True
                slider.update_value_from_mouse(mx)
                if self.callbacks.get("filter_value_change"):
                    self.callbacks["filter_value_change"](name, slider.value)
                return

        # Check record/export buttons.
        for btn in [self.btn_record, self.btn_export_wav, self.btn_export_png]:
            if btn.contains(mx, my):
                btn.pressed = True
                if btn.on_click:
                    btn.on_click()
                return

        # Check species panel clicks.
        species_panel = LAYOUT["species"]
        if species_panel["y"] <= my <= species_panel["y"] + species_panel["h"]:
            self._handle_species_click(mx, my)

    def handle_mouse_motion(self, mx: int, my: int) -> None:
        """Process a mouse-motion event (hover updates, slider dragging)."""
        self._mouse_x = mx
        self._mouse_y = my

        # Update hover states for all buttons.
        all_buttons = [self.btn_play, self.btn_pause, self.btn_stop,
                       self.btn_load, self.btn_mic,
                       self.btn_time_exp, self.btn_heterodyne, self.btn_freq_div,
                       self.btn_record, self.btn_export_wav, self.btn_export_png]
        all_buttons.extend(self.fft_buttons)

        for btn in all_buttons:
            btn.hover = btn.contains(mx, my)

        # Handle slider dragging.
        if self.slider_method.dragging:
            self.slider_method.update_value_from_mouse(mx)
            if self.callbacks.get("method_value_change"):
                self.callbacks["method_value_change"](self.slider_method.value)

        for slider in self.filter_sliders.values():
            if slider.dragging:
                slider.update_value_from_mouse(mx)
                # Determine which parameter to update based on filter name.
                param_map = {
                    "highpass": "cutoff", "lowpass": "cutoff",
                    "bandpass": "center", "noise_gate": "threshold",
                    "gain": "gain",
                }
                if self.callbacks.get("filter_value_change"):
                    name = [k for k, v in self.filter_sliders.items() if v is slider][0]
                    self.callbacks["filter_value_change"](name, slider.value)

    def handle_mouse_up(self, mx: int, my: int) -> None:
        """Process a mouse-up event (release)."""
        self._mouse_down = False

        # Release all button pressed states.
        all_buttons = [self.btn_play, self.btn_pause, self.btn_stop,
                       self.btn_load, self.btn_mic,
                       self.btn_time_exp, self.btn_heterodyne, self.btn_freq_div,
                       self.btn_record, self.btn_export_wav, self.btn_export_png]
        all_buttons.extend(self.fft_buttons)

        for btn in all_buttons:
            btn.pressed = False

        # Release slider dragging.
        self.slider_method.dragging = False
        for slider in self.filter_sliders.values():
            slider.dragging = False

    def _select_method(self, method_key: str) -> None:
        """Handle method selector button clicks."""
        # Update active states.
        self.btn_time_exp.active = (method_key == "time_expansion")
        self.btn_heterodyne.active = (method_key == "heterodyne")
        self.btn_freq_div.active = (method_key == "frequency_division")

        # Update the method control slider's range and label.
        if method_key == "time_expansion":
            self.slider_method.min_val = 1
            self.slider_method.max_val = 20
            self.slider_method.value = 10
            self.slider_method.label = "Expansion Factor (1x–20x)"
            self.slider_method.integer = False
        elif method_key == "heterodyne":
            self.slider_method.min_val = 20
            self.slider_method.max_val = 120
            self.slider_method.value = 40
            self.slider_method.label = "Oscillator Frequency (20–120 kHz)"
            self.slider_method.integer = True
        elif method_key == "frequency_division":
            self.slider_method.min_val = 2
            self.slider_method.max_val = 16
            self.slider_method.value = 10
            self.slider_method.label = "Divisor (2, 4, 8, 10, 16)"
            self.slider_method.integer = True

        if self.callbacks.get("method_change"):
            self.callbacks["method_change"](method_key)

    def _handle_species_click(self, mx: int, my: int) -> None:
        """Handle a click in the species reference panel."""
        from species_guide import SPECIES_DATABASE

        species_x = 50
        species_y = LAYOUT["species"]["y"] + 25  # Below the panel title
        row_h = 22

        for i, species in enumerate(SPECIES_DATABASE):
            row_y = species_y + i * row_h
            if row_y <= my <= row_y + row_h and species_x <= mx <= species_x + 700:
                self._selected_species_idx = i
                # Click-to-tune: set the heterodyne oscillator to this species' frequency.
                if self.callbacks.get("species_select"):
                    self.callbacks["species_select"](species.peak_freq_khz)
                return

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, spectrogram_surf: pygame.Surface | None = None,
             is_playing: bool = False, is_recording: bool = False,
             audio_info_text: str = "", status_text: str = "",
             shifted_range_text: str = "", mic_active: bool = False) -> None:
        """
        Render the complete UI on the screen.

        Args:
            screen:            The main pygame display surface.
            spectrogram_surf:  The rendered spectrogram surface to blit, or None.
            is_playing:        Whether audio is currently playing.
            is_recording:      Whether recording is active.
            audio_info_text:   File/mic info text for the transport bar.
            status_text:       Status bar text.
            shifted_range_text: Text showing the shifted frequency range.
            mic_active:        Whether the live mic is active.
        """
        # Clear screen with background color.
        screen.fill(COLORS["bg"])

        # --- Title Bar ---
        title = self._title_font.render("BAT LISTENER — Ultrasonic Audio Translator", True, COLORS["accent"])
        screen.blit(title, (CANVAS_WIDTH // 2 - title.get_width() // 2, 12))

        # --- Transport Bar ---
        # Update play/pause button active state.
        self.btn_play.active = is_playing and not is_recording
        self.btn_mic.active = mic_active

        self.btn_play.draw(screen, self._font)
        self.btn_pause.draw(screen, self._font)
        self.btn_stop.draw(screen, self._font)
        self.btn_load.draw(screen, self._font)
        self.btn_mic.draw(screen, self._font)

        # File info text.
        info_surf = self._small_font.render(audio_info_text, True, COLORS["text"])
        screen.blit(info_surf, (460, LAYOUT["transport"]["y"] + 18))

        # Progress bar.
        prog_y = LAYOUT["transport"]["y"] + 42
        prog_x = 50
        prog_w = CANVAS_WIDTH - 100
        pygame.draw.rect(screen, COLORS["slider_bg"], (prog_x, prog_y, prog_w, 4))
        fill_w = int(self._progress * prog_w)
        pygame.draw.rect(screen, COLORS["accent"], (prog_x, prog_y, fill_w, 4))
        time_surf = self._tiny_font.render(self._time_text, True, COLORS["text_dim"])
        screen.blit(time_surf, (prog_x, prog_y - 12))

        # --- Method Selector ---
        self.btn_time_exp.draw(screen, self._font)
        self.btn_heterodyne.draw(screen, self._font)
        self.btn_freq_div.draw(screen, self._font)

        # --- Method Controls ---
        self.slider_method.draw(screen, self._font, self._small_font)

        # FFT size label + buttons.
        fft_label = self._small_font.render("FFT:", True, COLORS["text"])
        screen.blit(fft_label, (500, LAYOUT["controls"]["y"] + 4))
        for btn in self.fft_buttons:
            btn.draw(screen, self._small_font)

        # Shifted frequency range text.
        range_surf = self._small_font.render(shifted_range_text, True, COLORS["accent"])
        screen.blit(range_surf, (50, LAYOUT["controls"]["y"] + 42))

        # --- Spectrogram Panel ---
        spec_y = LAYOUT["spectrogram"]["y"]
        # Draw panel border.
        pygame.draw.rect(screen, COLORS["border"],
                         (45, spec_y - 5, CANVAS_WIDTH - 90, LAYOUT["spectrogram"]["h"] + 10), 2)

        # Blit the spectrogram surface if available.
        if spectrogram_surf is not None:
            # Scale to fit the panel.
            scaled = pygame.transform.scale(spectrogram_surf,
                                            (CANVAS_WIDTH - 94, LAYOUT["spectrogram"]["h"]))
            screen.blit(scaled, (47, spec_y - 3))
        else:
            # Placeholder text.
            placeholder = self._font.render("[ Spectrogram — load a file or start mic ]", True, COLORS["text_dim"])
            screen.blit(placeholder, (CANVAS_WIDTH // 2 - placeholder.get_width() // 2, spec_y + 80))

        # --- Species Panel ---
        self._draw_species_panel(screen)

        # --- Filter Panel ---
        self._draw_filter_panel(screen)

        # --- Record / Export Bar ---
        self.btn_record.active = is_recording
        if is_recording:
            self.btn_record.label = "● STOP REC"
            self.btn_record.color = COLORS["error"]
        else:
            self.btn_record.label = "● RECORD"
            self.btn_record.color = COLORS["accent"]

        self.btn_record.draw(screen, self._font)
        self.btn_export_wav.draw(screen, self._font)
        self.btn_export_png.draw(screen, self._font)

        # Recording timer.
        if is_recording and self.callbacks.get("get_rec_time"):
            rec_time = self.callbacks["get_rec_time"]()
            timer_surf = self._font.render(f"REC: {rec_time}", True, COLORS["error"])
            screen.blit(timer_surf, (430, LAYOUT["record"]["y"] + 8))
        else:
            ready_surf = self._font.render("READY", True, COLORS["text_dim"])
            screen.blit(ready_surf, (430, LAYOUT["record"]["y"] + 8))

        # --- Status Bar ---
        status_y = LAYOUT["status"]["y"]
        status_surf = self._small_font.render(status_text, True, COLORS["text"])
        screen.blit(status_surf, (50, status_y + 8))

        # Match indicator (right side of status bar).
        if self._match_text:
            match_color = COLORS["active"] if "match" in self._match_text.lower() else COLORS["text_dim"]
            match_surf = self._small_font.render(self._match_text, True, match_color)
            screen.blit(match_surf, (CANVAS_WIDTH - match_surf.get_width() - 20, status_y + 8))

    def _draw_species_panel(self, screen: pygame.Surface) -> None:
        """Draw the bat species reference panel with the match indicator."""
        from species_guide import SPECIES_DATABASE, find_species_match

        panel = LAYOUT["species"]
        px, py = 45, panel["y"]
        pw, ph = CANVAS_WIDTH - 90, panel["h"]

        # Panel background and border.
        pygame.draw.rect(screen, COLORS["panel_bg"], (px, py, pw, ph))
        pygame.draw.rect(screen, COLORS["border"], (px, py, pw, ph), 1)

        # Panel title.
        title = self._font.render("Species Guide — click to tune heterodyne oscillator", True, COLORS["accent"])
        screen.blit(title, (px + 10, py + 5))

        # Species rows.
        row_h = 22
        start_y = py + 25
        for i, species in enumerate(SPECIES_DATABASE):
            row_y = start_y + i * row_h

            # Highlight matched species in green.
            is_match = (i == self._match_species_idx)
            is_selected = (i == self._selected_species_idx)

            if is_match:
                bg = (0, 80, 40)  # Dark green for match
            elif is_selected:
                bg = (0, 60, 80)  # Dark cyan for selected
            else:
                bg = COLORS["panel_bg"] if i % 2 == 0 else (20, 20, 40)

            pygame.draw.rect(screen, bg, (px + 5, row_y, pw - 10, row_h - 2))

            # Species name + frequency.
            name_text = f"{species.name} ({species.peak_freq_khz:.0f} kHz)"
            color = COLORS["active"] if is_match else COLORS["text"]
            name_surf = self._small_font.render(name_text, True, color)
            screen.blit(name_surf, (px + 10, row_y + 4))

            # Call type + rhythm (right side).
            detail = f"{species.call_type}, {species.rhythm}"
            detail_surf = self._tiny_font.render(detail, True, COLORS["text_dim"])
            screen.blit(detail_surf, (px + pw - detail_surf.get_width() - 15, row_y + 6))

    def _draw_filter_panel(self, screen: pygame.Surface) -> None:
        """Draw the audio filter panel with toggles and sliders."""
        panel = LAYOUT["filters"]
        px, py = 45, panel["y"]
        pw, ph = CANVAS_WIDTH - 90, panel["h"]

        # Panel background and border.
        pygame.draw.rect(screen, COLORS["panel_bg"], (px, py, pw, ph))
        pygame.draw.rect(screen, COLORS["border"], (px, py, pw, ph), 1)

        # Panel title.
        title = self._small_font.render("Audio Filters & Enhancement", True, COLORS["accent"])
        screen.blit(title, (px + 10, py + 2))

        # Draw each filter's toggle + slider.
        for name, toggle in self.filter_toggles.items():
            toggle.draw(screen, self._small_font)
            slider = self.filter_sliders[name]
            slider.draw(screen, self._small_font, self._tiny_font)

    # -----------------------------------------------------------------------
    # State Updates (called from main loop)
    # -----------------------------------------------------------------------

    def update_progress(self, progress: float, time_text: str) -> None:
        """Update the progress bar and time display."""
        self._progress = max(0.0, min(1.0, progress))
        self._time_text = time_text

    def update_match(self, match_text: str, species_idx: int | None = None) -> None:
        """Update the species match indicator."""
        self._match_text = match_text
        self._match_species_idx = species_idx

    def update_method_display(self, method: str, value: float) -> None:
        """Update the method selector and slider to reflect current state."""
        self.btn_time_exp.active = (method == "time_expansion")
        self.btn_heterodyne.active = (method == "heterodyne")
        self.btn_freq_div.active = (method == "frequency_division")
        self.slider_method.value = value

    def update_filter_display(self, filters: dict) -> None:
        """Update filter toggles and sliders to reflect current state."""
        param_map = {
            "highpass": "cutoff", "lowpass": "cutoff",
            "bandpass": "center", "noise_gate": "threshold",
            "gain": "gain",
        }
        for name, toggle in self.filter_toggles.items():
            f = filters.get(name)
            if f:
                toggle.active = f.active
                param = param_map.get(name, "cutoff")
                if hasattr(f, param):
                    self.filter_sliders[name].value = getattr(f, param)